import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import json

login(token="hf_token")

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"

)

# Potential way of handling long documents: check every chunk
def chunk_text(text, chunk_size=4000):

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):
        chunks.append(
            text[i:i+chunk_size]
        )

    return chunks





def find_survey_sections(chunk):

    prompt = f"""
You are analyzing an academic paper.

Determine whether this text contains survey questions.

Text:

{chunk}


Return JSON:

{{
"contains_survey_information": true/false,
"relevant_excerpt": ""
}}
"""

    return generate_response(prompt)

def generate_response(prompt):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0,
        do_sample=False
    )

    generated = outputs[0][inputs.input_ids.shape[1]:]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return response


def extract_survey_questions(section_text):

    prompt = f"""
You are extracting survey instruments from an academic paper.

Extract any survey questions, questionnaire items,
constructs, and response scales.

Text:

{section_text}

Return JSON only:

{{
"survey_items": [],
"constructs": [],
"response_scales": []
}}
"""

    response = generate_response(prompt)

    try:
        return json.loads(response)

    except:
        return {
            "survey_items": [],
            "constructs": [],
            "response_scales": [],
            "raw_response": response
        }


def main():



# Hardcoding text for now, will later use pdfplumber or similar to read text
    paper_text = """
    MILITARY MEDICINE, 183, 11/12:e518, 2018
The Voice of the Consumer: A Survey of Veterans and Other Users
of Assistive Technology
Brad E. Dicianno, MD*†‡; James Joseph, Gunnery Sergeant USMC (RET), MS*;
Stacy Eckstein, BS, MT (ASCP)*; Christina K. Zigler, PhD, MSEd*†; Eleanor Quinby, BS†;
Mark R. Schmeler, PhD, OTR/L, ATP‡; Richard M. Schein, PhD, MPH‡; Jon Pearlman, PhD*‡;
Rory A. Cooper, PhD*†‡
ABSTRACT Introduction: A total of 3.6 million Americans and over 250,000 veterans use wheelchairs. The need
for advancements in mobility-assistive technologies is continually growing due to advances in medicine and rehabilita
tion that preserve and prolong the lives of people with disabilities, increases in the senior population, and increases in
the number of veterans and civilians involved in conflict situations. The purpose of this study is to survey a large sam
ple of veterans and other consumers with disabilities who use mobility-assistive technologies to identify priorities for
future research and development. Materials and Methods: This survey asked participants to provide opinions on the
importance of developing various mobility-assistive technologies and to rank the importance of certain technologies.
Participants were also asked to provide open-ended comments and suggestions. Results: A total of 1,022 individuals,
including 500 veterans, from 49 states within the USA and Puerto Rico completed the survey. The average age of
respondents was 54.3 yr, and they represented both new and experienced users of mobility-assistive technologies. The
largest diagnostic group was spinal cord injury (SCI) (N = 491, 48.0%). Several themes on critical areas of research
emerged from the open-ended questions, which generated a total of 1,199 comments. Conclusion: This survey revealed
several themes for future research and development. Advanced wheelchair design, smart device applications, human
machine interfaces, and assistive robotics and intelligent systems emerged as priorities. Survey results also demon
strated the importance for researchers to understand the effects of policy and cost on translational research and to be
involved in educating both consumers and providers.
INTRODUCTION
A total of 3.6 million Americans1 and over 250,000 veterans use
wheelchairs. (Personal communications among Penny Nechanicky
[National Director], Ru Gakhar [Prosthetic Program Manager],
Prosthetic and Sensory Aids Service, Department of Veterans
Affairs, Rory Cooper, and Brad Dicianno. July 24, 2015.)
Mobility-assistive technologies such as wheelchairs and scoo
ters improve function, home and community integration,2–5
quality of life,6,7 and comfort.6,8 The number of users of these
technologies is continually growing due to advances in medi
cine and rehabilitation that preserve and prolong the lives of
people with disabilities, increases in the senior population, and
increases in the number of veterans and civilians involved in
conflict situations.1,9,10
*Human Engineering Research Laboratories, VA Pittsburgh Healthcare
System, Pittsburgh, PA 15206.
†Department of Physical Medicine and Rehabilitation, School of
Medicine, University of Pittsburgh, Pittsburgh, PA 15213.
‡Department of Rehabilitation Science and Technology, School of
Health and Rehabilitation Sciences, University of Pittsburgh, Pittsburgh, PA
15260.
The contents of this publication do not represent the views of the
Department of Veterans Affairs or the United States Government.
doi: 10.1093/milmed/usy033
Published by Oxford University Press on behalf of the Association of
Military Surgeons of the United States 2018. This work is written by (a) US
Government employee(s) and is in the public domain in the US.
e518
Despite the availability of mobility-assistive technologies to
the various populations in need, The World Health Organization
(WHO) estimates that over 1 billion people currently need
assistive technologies, but that only 1 in 10 have access. They
estimate that by 2050, due to the growth of disability popula
tions, 2 billion people will be in need.11 In cooperation with
the United States Agency for International Development
(USAID), the United Nations, and other agencies, WHO estab
lished the Global Cooperation on Assistive Technology (GATE),
which is a global effort to improve access to high-quality afford
able assistive products.11 In order to reach the goal of global
access, a research agenda for assistive technologies must first be
developed.
The need for additional support for mobility is also a top pri
ority of veterans with disabilities. A report by Paralyzed
Veterans of America (PVA) named mobility and independence
among the top three health concerns of female veterans who
were injured or diagnosed with a neurological condition.12 We
are not aware of a similar study conducted on men. Simpson
et al. published a systematic review13 that summarized 24 stud
ies (5,262 total participants with spinal cord injury [SCI]) that
directly surveyed individuals with SCI about their health and
function priorities. This review revealed that within the health
domain, motor function, bowel and bladder function, and sex
ual function were the top three priorities. Those with paraplegia
prioritized mobility, whereas those with tetraplegia prioritized
hand function. General health and relationships were also cited
Downloaded from https://academic.oup.com/milmed/article/183/11-12/e518/4959951 by Purdue University user on 25 March 2026
MILITARY MEDICINE, Vol. 183, November/December 2018
The Voice of the Consumer: A Survey of Veterans and Other Users of Assistive Technology
as priorities. Another systematic review of six studies (1,222
total participants with SCI) revealed that mobility was a top
concern after SCI. Research and development to advance
mobility-assistive technologies could begin to address these
concerns.
In two hearings before Subcommittees of the Appropriations
Committee of the House of Representatives, testimony was
given by high ranking government officials about the importance
of research on mobility-assistive technologies. Vice Admiral
Matthew Nathan, the 37th Surgeon General of the Navy and
Chief of the Navy’s Bureau of Medicine and Surgery, stressed
the need to continue funding for research specifically geared
to improve the mobility of injured veterans.14 Robert A.
MacDonald, Secretary, Department of VA, requested support
of Congress for Veteran health care and research.15 Additionally,
in testimony before the Subcommittee on Social Security,
Committee on Ways and Means, Robert E. Robertson, Director,
Education, Workforce, and Income Security Issues, specifically
mentioned wheelchair design as a scientific advancement that
has allowed individuals with disabilities to participate in society,
including seeking and maintaining employment.16 The Social
Security Administration, with encouragement from the Office of
Management and Budget, commissioned the National Academy
of Medicine (NAM) to conduct a study on advances in wheel
chairs and other mobility devices on employability of people
with disabilities.
Because of this growing need for better technology, research
aims must be outlined and prioritized. The National Academies
of Sciences, Engineering, and Medicine (NAS), in collabora
tion with the Social Security Administration, provided a
report analyzing the use of current assistive technologies and
called for development of research priorities.17 The President’s
Council of Advisors on Science and Technology (PCAST)
report recommended that the VA and Centers for Medicare
and Medicaid Services create a “road map” for mobility-assistive
technology research and that more federal support of research
be made available.18 However, this road map has yet to be
developed.
We began to develop such a road map in our preliminary
work. We conducted a pilot study of 112 individuals who
use mobility-assistive technologies to characterize their
needs and opinions.19 In parallel, we also surveyed 161 pro
fessionals involved in the provision of mobility devices both
within and outside VA.31 These surveys revealed several
themes that helped us to construct a preliminary road map,
but a larger and more diverse sample of users was needed.
The purpose of the present study was to evaluate the opi
nions of over 1,000 users of mobility-assistive technologies
to inform a research agenda and identify priorities that are
aligned with the goals and desires of the users.
METHODS
The survey (see Appendix 1) was developed by experts in
the field of assistive technology at the Human Engineering
Research Laboratories (HERL) in Pittsburgh, PA, USA. The VA
Pittsburgh Healthcare System Veterans Engineering Resource
Center (VERC) assisted the team in establishing content valid
ity. The study was approved by the Institutional Review Board
of the University of Pittsburgh as an expedited study. The sur
vey was administered through the Research Electronic Data
Capture (REDCap) system (Vanderbilt University, Nashville,
TN, USA), a secure, web-based software system sponsored by
the Clinical Translational and Science Institute at the University
of Pittsburgh.
The survey was designed to take less than 10 min to com
plete and was based on a previously conducted survey.19
Participants could complete the survey online using a com
puter or any mobile device. Alternately, they had the option
to contact a study coordinator to complete the questionnaire
by phone or mail.
The home page provided a brief description of the project
and an overview of informed consent, asking the participant
to answer whether they consented to taking the survey.
Participants were asked to provide basic demographic infor
mation and to choose from a list of pre-defined diagnoses or
list their own diagnosis under “other.” Participants were
allowed to list multiple diagnoses. Some personally identifi
able information was collected in the questionnaire to ensure
that there were no repeat responses; however, the question
naire was anonymized by the coordinator prior to analysis.
Participants were asked how important it is to carry out
certain activities if technology could accommodate them.
The next set of questions asked participants to provide indi
vidual rankings of importance of developing various technol
ogies that were ranked highly in our previous survey.19 The
survey also required participants to rank these items against
each other in terms of order of importance. Participants were
asked how often they play an active role in the decision
making process when getting new mobility equipment and
how often they receive adequate support to be able to main
tain their assistive technology long term. Participants were
also asked to respond to several open-ended questions or
statements, including identifying barriers when obtaining
new mobility-assistive technologies.
Inclusion criteria were age 18 yr or older, U.S. citizen, an
individual who uses mobility-related assistive devices, and
answers “yes” to providing consent to participating in the
survey. There were no specific exclusion criteria.
Referral sampling was used for this study wherein initial
participants were asked to distribute recruitment materials to
their own networks. Participants were recruited in person at ath
letic events, advocacy meetings, and at meetings of veteran ser
vice organizations for veterans and people with disabilities.
Flyers were sent via email to personal contacts and professional
listservs at disability and veteran service organizations, veteran
support groups, health care agencies, hospitals, clinics, govern
ment organizations, and universities. Personal contacts then dis
tributed recruitment materials through social media and email
outreach. Flyers were posted on our own organization’swebsite
Downloaded from https://academic.oup.com/milmed/article/183/11-12/e518/4959951 by Purdue University user on 25 March 2026
MILITARY MEDICINE, Vol. 183, November/December 2018
e519
The Voice of the Consumer: A Survey of Veterans and Other Users of Assistive Technology
and social media pages. Participants were also recruited from
our local research registries, which contain contact information
of individuals who have expressed interest in participating in
research. Recruitment materials were advertised in magazines,
targeted social media ads, and newsletters and also mentioned
in radio broadcasts. Potential participants were directed to
access the web link directly or to contact a study coordinator.
Descriptive statistics for each multiple-choice item were
reported using frequency counts and percentages. Average
rankings were examined for sets of items that were placed in
order by respondents. For respondents with SCI, a sub-analysis
was performed to compare differences in the research and
development priorities of those with paraplegia versus tetraple
gia. Chi-squared tests of independence were used to determine
the relationship between the rankings of each item and group
membership. All descriptive statistics were performed using
SPSS v24.0 (IBM Corp., Armonk, NY, USA). Open-ended
responses were examined in detail to identify overall patterns
and themes, as well as unique, creative solutions to the issues
of mobility.
RESULTS
A total of 1,127 individuals were sent recruitment materials
and asked to distribute further to their own contacts. Also, tar
geted social media ads reached 41,129 unique people. Referral
sampling resulted in 1,178 potential participants. Figure 1 dis
plays an exclusion diagram demonstrating inclusion of 1,022
participants based on aforementioned criteria. Table I displays
demographic characteristics of participants. Average age was
54.3 (STD 14.6, range 19–95) yr. A total of 500 (48.9%) were
veterans. Participants lived in 49 different states within the U.
S. and Puerto Rico. Four individuals required help from a clin
ical coordinator to complete the survey via phone and one via
mail; the rest completed the survey electronically.
Supplemental Table 1 displays diagnoses of participants.
The largest diagnostic group was spinal cord injury (SCI) (N =
491, 48.0%). Of the participants with SCI, 290 (59.1%) had
paraplegia, 188 (38.3%) had tetraplegia, and 13 (2.6%) did not
report this classification. A total of 212 (43.2%) reported com
plete SCI, 252 (51.3%) reported incomplete SCI, and 27 (5.5%)
did not report completeness of lesion. Some individuals indic
ating “other” diagnoses were reclassified into one of the pre
defined categories if clinically appropriate.
When asked about how important it would be to carry out
certain activities if technology could accommodate them, the
majority of respondents identified all four categories of activ
ities as critical/important (Supplemental Table 1). Participants
were also asked how important it would be for researchers to
develop specific technologies. Although most technologies
were rated as critical or important, wheelchairs and compo
nents that could self-adjust or could assist in overcoming
obstacles gained the most critical ratings (Table II).
When asked to rank four areas of technology development,
five “futuristic inventions” and four “futuristic mobility/
transportation inventions” from least to most important
using a 4- or 5-point scale (Tables III–V), smart wheelchair
design, transfer devices, smart home technology, exoskeletons,
and new power sources for wheelchairs received the highest
rankings.
Less than half of the participants (n = 471, 46.1%) felt
that people with disabilities often (n = 349, 34.1%) or always
(n = 122, 11.9%) play an active role in the decision-making
process when getting new mobility equipment. Furthermore,
the majority (n = 647, 63.3%) also said that people with dis
abilities rarely (n = 572, 56.0 %) or never (n = 75, 7.3%)
receive adequate support to be able to maintain their assistive
technology long term.
Similar responses were seen between those with tetraplegia
and paraplegia except for two distinct instances. More partici
pants with tetraplegia ranked human–machine interfaces as
important or most important than those with paraplegia (47.3%
versus 33.4%, respectively; p = 0.017). More participants with
paraplegia ranked a “manual wheelchair that would fold or dis
assemble” as important or most important than those with tet
raplegia (49.6% versus 33.5%, respectively; p < 0.001).
The most commonly cited barrier to obtaining new mobility
assistive technologies was the funding and procurement pro
cess, particularly cost, followed by knowledge of the user and
the provider (Supplemental Table 3).
Several themes on critical areas of research emerged from
open-ended questions that generated a total of 1,199 comments
(Supplemental Table 4). Additionally, 73 individuals provided
more information about their own personal disability or medical
condition and 16 provided comments about ways to improve
the survey.
DISCUSSION
This survey of over 1,000 consumers of mobility-assistive technol
ogies can be used to develop a research and development “road
map” that has been called for by PCAST and NAS. Consumers
felt that mobility technology is important for all aspects of
their lives. Participants also emphasized the importance of
research and development on mobility-assistive technologies
and the need to include people with disabilities in research
and development. This latter concept is nicely summarized
by two insightful quotes from participants:
• “A futurist once told me that tech design should include
disabled people the same way the military includes test
pilots. Test pilots are trained for their job evaluating
planes and they are, mostly, listened to.”
• “Clinicians and researchers need to listen to veterans and
people with disabilities. They should not make assump
tions as to what is important. People with disabilities are
smart too.”
Open-ended responses revealed an opportunity for improv
ing dissemination and education pathways to teach consumers
Downloaded from https://academic.oup.com/milmed/article/183/11-12/e518/4959951 by Purdue University user on 25 March 2026
e520
MILITARY MEDICINE, Vol. 183, November/December 2018
The Voice of the Consumer: A Survey of Veterans and Other Users of Assistive Technology
Individuals who were presented the
survey: (n = 1178)
Individuals who consented after reading
informational script: (n = 1157)
Subjects who did not consent after
reading the informational script: (n = 21)
Entries excluded because of duplicates:
(n = 109)
Number of non-duplicate entries:
(n = 1048)
Total number of Eligible subjects:
(n = 1022)
FIGURE 1. Exclusion flow chart.
Other Exclusions: (n = 26)
Justifications for other Exclusions
• Individuals provided either no date of birth or ineligible birth
date: (n = 16)
• Individuals stated that they completed the survey from outside the
US: (n = 5)
• Individuals provided no state of residence and we were not able to
prove or reasonably assume that they lived in the US: (n = 2)
• One individual recorded three responses which were all inconsistent;
two were excluded as duplicates above, and the third was removed
here: (n = 1)
• Individual completed the survey for a deceased relative: (n = 1)
• Survey was from an individual testing the survey before
distributing: (n = 1)
about advances in assistive technology research. Some partici
pants recommended developing products that are already on
the market, suggesting that they were unaware of their availability.
Other participants suggested specific ways consumers could be
educated about technologies on the market, including expos,
websites, or other tools. Participants felt that both consumers
and providers needed education.
Open-ended responses also emphasized the need for new
and better technology but at lower cost. Participants felt that
cost was a barrier to obtaining new devices, insofar as it affected
availability of funding. The process of obtaining equipment was
in many cases identified as laborious and inefficient. This places
a responsibility on researchers to mitigate cost when developing
devices and understanding how insurance policies may affect
translation of the technology into the hands of consumers.
Based on the results of the survey, a conceptual framework
for mobility-assistive technology research and development was
produced (Fig. 2). The process is person-centered and should be
conducted with an understanding of the broader concepts of
universal design, policy, clinical practice, and cost. Four
research thrust areas represent mobility-assistive technology
research and development priorities. Education, dissemination
and knowledge transfer, and standards and reliability are criti
cal outputs that must occur alongside the development and
clinical testing of mobility-assistive technology. Not repre
sented in this figure are the views regarding other research
domains, such as regenerative medicine and devices to assist
self-management. As this survey was specifically designed to
elicit feedback on technology used for mobility, more in
depth surveys would be needed to develop similar frame
works for other domains. Some of this work has been
reported elsewhere.20,21
All four research thrust areas identified for mobility-assistive
technology have been noted as opportunities for rehabilitation
research in an expert report published by the US Department of
Veterans Affairs Office of Research and Development.22 The
first thrust is “advanced wheelchair design.” Participants placed
emphasis on technology that can avoid collisions or help to
Downloaded from https://academic.oup.com/milmed/article/183/11-12/e518/4959951 by Purdue University user on 25 March 2026
MILITARY MEDICINE, Vol. 183, November/December 2018
e521
negotiateobstacles; lighterweight, folding,or smallerwheel
chairs;andalternativepowersourcesforwheelchairs.Maneu
verabilityandtransportabilitywereseenascriticalformobility
inthehome, inthecommunity,andduringtravel.Participants
expressedafrustrationwithcurrent transportation,adesirefor
expandedoptionsinthefieldofaccessibledriving,andaneed
forchangeinairlinepoliciesandaccessibility.Therequestsfor
alternativepowersourceswasnotsurprising,giventhatbatter
iesandelectrical componentsare themost commoncompo
nentstofailandneedreplacement.23,24
Second, “smart deviceapplications” that consumers can
use in their home orwear are needed tohelpusers track
information and control their environments. Participants
wereparticularly interested insmart home technology.Our
ownresearchonmonitoringandcoaching technologieshas
demonstratedwaysthatwearabledevicesandcoachingtech
nologiescanbeusedbyindividualswithdisabilities topro
motehealthandphysicalactivity.25
A third notable thrust is “human–machine interfaces.”
Participants with limited armor handmovement wanted
alternativeways tocontrolwheelchairs using thevoiceor
face.Ourownresearch in thisareahas focusedoncontrol
strategies thataresharedbetweentheuserandthedevice,26
universal interfaces that can controlmultiple devices, and
bettersoftwarealgorithms.27,28
Finally, “assistive robotics and intelligent systems”were
seenas ahighpriority. Participantshighlighted theneed for
self-drivingwheelchairsandnavigationassistance,betterexo
skeleton technology for ambulation, and devices that assist
withtransfersofpeopleordevicesintoandoutofvehicles,or
that transferpeople intoandout ofwheelchairs.Weplan to
address this thrust by investigatinghownavigation, sensing,
andcontrol systems canadapt to theneedsof theuser and
howtheycanlearnfromtheuser.28,29Wealsoplantoadvance
thefieldbydevelopingdevicesthatassistwithtransferstoand
fromwheelchairsandbeds30andthataimtoimprovesafetyof
caregiverswhoperformthetransfers.25,28
Consumers tended to agreewith providers of technol
ogy31withafewexceptions.Consumersplacedpriorityon
thesametechnologiesasproviders(devicesthataidtransfers
andalternativepowersourcesforwheelchairs),but theyalso
emphasizeda fewmore: smart home technology, exoskele
tons,andsmartwheelchairdesign.Themajorityofproviders
perceived that their clientsplayanactive role inobtaining
their assistive technologies.Themajorityof consumers, on
theotherhand, felt theyrarelyorneverplayanactiverole.
Thismismatch inperceptions suggests that providersmay
TABLEI. DemographicInformationforConsumers(n=1,022)
n %
Typesofassistivedevicesused(participantscould
choosemorethanone)
Manualwheelchair 596 58.3
Powerwheelchair 481 47.1
Scooter 98 9.6
Lowerextremityprosthesis 49 4.8
Lowerextremityorthosis(brace) 134 13.1
Assistivedevice(e.g.,cane,crutch,andwalker) 387 37.9
Other 99 9.7
Lengthoftimedevice(s)used
1yrorless 58 5.7
2–5yr 235 23.0
6–10yr 168 16.4
11–15yr 135 13.2
Morethan15yr 423 41.4
Missing/didnotanswer 3 <1.0
Gender
Female 367 35.9
Male 655 64.1
Highest levelofeducation
Associate’sdegree 218 21.3
Bachelor’sdegree 284 27.8
Doctorateleveldegree–MD,DO,PhD 60 5.9
Highschooldiplomaorequivalent(GED) 245 24.0
Master’sdegree 178 17.4
Otheradvanceddegree 36 3.5
Missing 1 <1.0
VeteranofUSArmedForces
No 520 50.9
Yes 500 48.9
Missing 2 <1.0
Typeofcommunitysettinginwhichyoulive
Rural(country) 232 22.7
Suburban 460 45.0
Urban(city) 322 31.5
Missing 8 <1.0
Ethnicity
HispanicorLatino 63 6.2
NotHispanicorLatino 954 93.3
Missing 5 <1.0
Race
White/Caucasian 841 82.3
BlackorAfricanAmerican 78 7.6
Twoormoreraces 49 4.8
Other 32 3.1
AmericanIndianorAlaskanNative 9 <1.0
Asian 7 <1.0
NativeHawaiianorotherPacificIslander 4 <1.0
Missing 2 <1.0
Ifaveteran,obtainassistivedevicesthroughVA
Yes 439 87.8
No 59 11.8
Missing 2 <1.0
Householdincome
Under$15,000 115 11.3
$15,000–$24,999 109 10.7
$25,000–$49,999 195 19.1
$50,000–$74,999 150 14.7
$75,000–$100,000 117 11.4
Over$100,000 149 14.6
(continued)
TABLEI. Continued
n %
Idon’tknow 28 2.7
Iprefernot toanswer 158 15.5
Missing 1 <1.0
e522 MILITARYMEDICINE,Vol.183,November/December2018
TheVoiceof theConsumer:ASurveyofVeteransandOtherUsersofAssistiveTechnology
Downloaded from https://academic.oup.com/milmed/article/183/11-12/e518/4959951 by Purdue University user on 25 March 2026
need toplacemore emphasis onpatient-centeredcare and
inclusiveness.However, themajorityofprovidersandusers
(68.3%and63.3%,respectively)agreedthatpeoplewithdis
abilities rarelyornever receiveadequate long-termsupport
tomaintain their technologies.This suggests that access to
rehabilitation technicians and engineers and training pro
grams32 to teachconsumers andprovidershowtoperform
basicmaintenanceareneeded.
Thedifferencesseeninopinionsof thosewithtetraplegia
andparaplegiawereexpectedandlikelyreflect their respec
tivefunctionalneeds.Themajorityof individualswithpara
plegiausedmanualwheelchairsandwere thus interestedin
more compact or foldingmanualwheelchairs. Thosewith
tetraplegiaand lossof armfunctionweremore likely than
those with paraplegia and full use of their arms to be
interestedinhuman–machineinterfacestohelpthemcontrol
otherdevices.
Afewlimitations to this researchstudydeservediscus
sion. First, because this study involved completion of an
onlinesurvey,wemayhaveoversampledthosewhoaretech
nologically savvy or those who have Internet access.
However,wedidprovidealternatemeansforcompletingthe
survey.Second,wesampledonlyasmall proportionof the
individuals in theU.S.whousemobility-assistive technolo
gies.However, theparticipantsrangedinagefrom19to95
yr, usedavarietyof assistive technologies, andrepresented
98%ofU.S.statesandPuertoRico.Respondentsrepresented
awide range of experienceswith technology, from those
whowerenovicestothosewhousedtechnologyfor15yror
more.Wealsosampledalargepopulationofveterans.Third,
TABLEII. RankingofKeyAreasforResearchfromCritical tonotImportant,n(%)
Missing
Response Critical Important
Minor
Importance
Not
Important
Developportablepoweredtransferdevicesthatapersonwithadisabilitycoulduse
independently?
2(0.2) 416(40.7) 439(43) 100(9.8) 65(6.4)
Developsportorrecreationtechnologytohelpyoumeetyourfitnessorweight
lossgoals?
2(0.2) 375(36.7) 429(42) 160(15.7) 56(5.5)
Decreasetheamountoftimeit takestoprovidecustomizedwheelchair
components(i.e.,3Dprintedonsite)?
1(0.1) 331(32.4) 413(40.4) 201(19.7) 76(7.4)
Developwearableormobiletechnologiesthatcanprovidehealthorother
informationtousersortheirclinicians?
2(0.2) 273(26.7) 420(41.1) 243(23.8) 84(8.2)
Developwheelchairsandcomponentsthatcanself-adjustorcanassist in
overcomingobstacles
2(0.2) 513(50.2) 395(38.6) 86(8.4) 26(2.5)
TABLEIII. RankingFourAreasofTechnologyDevelopmentfromLeastImportant toMostImportant,n(%)
MissingResponse MostImportant Important SomewhatImportant LeastImportant
Wearableormobiletechnologies 49(4.8) 195(19.1) 219(21.4) 272(26.6) 287(28.1)
Human–machineInterfaces 40(3.9) 121(11.8) 279(27.3) 349(34.1) 233(22.8)
Smartwheelchairdesign 30(2.9) 515(50.4) 239(23.4) 157(15.4) 81(7.9)
Alternativepowersources 18(1.8) 155(15.2) 267(26.1) 209(20.5) 373(36.5)
TABLEIV. RankingFiveFuturisticInventionsfromLeastImportant toMostImportant,n(%)
Missing MostImportant Important Neutral SomewhatImportant LeastImportant
Personalrobotservant 46(4.5) 124(12.1) 156(15.3) 220(21.5) 248(24.3) 228(22.3)
Brainimplant 41(4) 83(8.1) 147(14.4) 242(23.7) 272(26.6) 237(23.2)
Smarthometechnology 45(4.4) 330(32.3) 353(34.5) 162(15.9) 108(10.6) 24(2.3)
Transferdevices 22(2.2) 402(39.3) 258(25.2) 165(16.1) 136(13.3) 39(3.8)
Virtualreality 17(1.7) 44(4.3) 91(8.9) 195(19.1) 215(21) 460(45)
TABLEV. RankingFourMobility/TransportationFuturisticInventionsfromLeastImportant toMostImportant,n(%)
Missing
Response
Most
Important Important
Somewhat
Important
Least
Important
Self-drivingorroboticpoweredwheelchair 49(4.8) 165(16.1) 224(21.9) 291(28.5) 293(28.7)
Newpowersourcesforwheelchairs 36(3.5) 313(30.6) 336(32.9) 236(23.1) 101(9.9)
Manualwheelchairthatwouldfoldordisassembletofit ina
suitcase
21(2.1) 187(18.3) 266(26) 262(25.6) 286(28)
Exoskeletonfordailymobility 15(1.5) 321(31.4) 177(17.3) 197(19.3) 312(30.5)
e523 MILITARYMEDICINE,Vol.183,November/December2018
TheVoiceof theConsumer:ASurveyofVeteransandOtherUsersofAssistiveTechnology
Downloaded from https://academic.oup.com/milmed/article/183/11-12/e518/4959951 by Purdue University user on 25 March 2026
The Voice of the Consumer: A Survey of Veterans and Other Users of Assistive Technology
FIGURE 2. Framework for mobility-assistive technology research and development.
participants can sometimes be over-enthusiastic about many
items in a survey if they feel strongly about a topic. To
address this potential bias in responses, we asked participants
to rank technologies against each in order to stratify the
importance of some devices over others. Fourth, responses
on surveys can be biased toward the organization’s mission,
especially if participants want to please the surveyor or feel
strongly about the topic. We therefore consulted with the
VERC to establish content validity, provided several open
ended questions to allow participants to provide feedback
and ideas not mentioned in the survey, and based the content
of the survey on responses provided by participants in our
prior work.19 Qualitative data from open-ended questions
was factored heavily into the conceptualization of the
research thrusts.
In order to ensure that results of this survey accurately
reflect current consumer needs, it should be repeated fre
quently. Future surveys will assess needs and opinions of
families and caregivers about mobility-assistive technology
research and will measure consumer and provider awareness
of available products, research outputs, and clinical practice
guidelines and whether they are being used. Such informa
tion should drive research and development projects and pro
gram priorities. We also anticipate that these findings will be
helpful in allowing consumer needs and wants to drive
e524
innovation in the field. The current study focused on parti
cipants of both veteran and civilian status, across a wide
range of diagnoses, and with broad demographic range. A
goal of future surveys should be to provide more specific
research to explore how priorities differ between unique sets
of individuals. Future researchers in the field of mobility
assistive technologies should take heed of one survey partici
pant’s comment: “The biggest challenge with technology is not
inventing or making it. It is making it reliable and simple
enough for everyone.”
CONCLUSION
This survey of consumers who use mobility-assistive technolo
gies can drive research and development priorities. Advanced
wheelchair design, human–machine interfaces, smart device
applications, and assistive robotics and intelligent systems are
top priorities for future research efforts. Survey results also
demonstrated the importance for researchers to understand the
effects of policy and cost on translational research and to be
involved in educating consumers and providers.
    """

    chunks = chunk_text(paper_text)

    relevant_chunks = []


    # find survey sections
    for i, chunk in enumerate(chunks):

        result = find_survey_sections(chunk)

        print("\nCHUNK", i)
        print(result)

        # later: parse JSON and only keep true chunks


        relevant_chunks.append(chunk)


    # extract questions
    for chunk in relevant_chunks:

        extraction = extract_survey_questions(chunk)

        print("\nEXTRACTION")
        print(extraction)



if __name__ == "__main__":
    main()
